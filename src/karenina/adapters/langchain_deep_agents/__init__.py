"""LangChain Deep Agents adapter for natively agentic evaluation.

This adapter provides AgentPort, LLMPort, and ParserPort implementations
using LangChain Deep Agents (create_deep_agent). It enables provider-agnostic
agentic evaluation with built-in planning, context management, and subagent
orchestration.

Requires: pip install deepagents langchain-mcp-adapters

Adapter classes:
    - DeepAgentsAgentAdapter: Agent loops via create_deep_agent with MCP support
    - DeepAgentsLLMAdapter: Simple LLM invocation via single-turn agent
    - DeepAgentsParserAdapter: Structured output parsing

Utilities:
    - DeepAgentsMessageConverter: Convert between unified Message and LangGraph types
    - check_deep_agents_available: Check if deepagents is installed
    - convert_mcp_to_tools: Convert MCPServerConfig to LangChain tools
    - extract_deep_agents_usage: Extract UsageMetadata from agent results
    - deep_agents_messages_to_raw_trace: Format messages as raw trace string
    - deep_agents_messages_to_trace_messages: Structured trace_messages with per-call usage
"""

from typing import TYPE_CHECKING, Any

__all__ = [
    "DeepAgentsAgentAdapter",
    "DeepAgentsLLMAdapter",
    "DeepAgentsParserAdapter",
    "DeepAgentsMessageConverter",
    "check_deep_agents_available",
    "convert_mcp_to_tools",
    "extract_deep_agents_usage",
    "deep_agents_messages_to_raw_trace",
    "deep_agents_messages_to_trace_messages",
    "wrap_deep_agents_error",
]

if TYPE_CHECKING:
    from karenina.adapters.langchain_deep_agents.agent import DeepAgentsAgentAdapter
    from karenina.adapters.langchain_deep_agents.availability import check_deep_agents_available
    from karenina.adapters.langchain_deep_agents.errors import wrap_deep_agents_error
    from karenina.adapters.langchain_deep_agents.llm import DeepAgentsLLMAdapter
    from karenina.adapters.langchain_deep_agents.mcp import convert_mcp_to_tools
    from karenina.adapters.langchain_deep_agents.messages import DeepAgentsMessageConverter
    from karenina.adapters.langchain_deep_agents.parser import DeepAgentsParserAdapter
    from karenina.adapters.langchain_deep_agents.trace import (
        deep_agents_messages_to_raw_trace,
        deep_agents_messages_to_trace_messages,
    )
    from karenina.adapters.langchain_deep_agents.usage import extract_deep_agents_usage


def __getattr__(name: str) -> Any:
    """Lazy import adapter classes to avoid circular imports."""
    _imports = {
        "DeepAgentsAgentAdapter": "karenina.adapters.langchain_deep_agents.agent",
        "DeepAgentsLLMAdapter": "karenina.adapters.langchain_deep_agents.llm",
        "DeepAgentsParserAdapter": "karenina.adapters.langchain_deep_agents.parser",
        "DeepAgentsMessageConverter": "karenina.adapters.langchain_deep_agents.messages",
        "check_deep_agents_available": "karenina.adapters.langchain_deep_agents.availability",
        "convert_mcp_to_tools": "karenina.adapters.langchain_deep_agents.mcp",
        "extract_deep_agents_usage": "karenina.adapters.langchain_deep_agents.usage",
        "deep_agents_messages_to_raw_trace": "karenina.adapters.langchain_deep_agents.trace",
        "deep_agents_messages_to_trace_messages": "karenina.adapters.langchain_deep_agents.trace",
        "wrap_deep_agents_error": "karenina.adapters.langchain_deep_agents.errors",
    }

    if name in _imports:
        import importlib

        module = importlib.import_module(_imports[name])
        return getattr(module, name)

    raise AttributeError(f"module 'karenina.adapters.langchain_deep_agents' has no attribute '{name}'")
