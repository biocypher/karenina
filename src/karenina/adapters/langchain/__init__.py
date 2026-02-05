"""LangChain adapter - wraps existing LangChain/LangGraph infrastructure.

This adapter provides implementations of the port interfaces (LLMPort, AgentPort,
ParserPort) using LangChain and LangGraph. It wraps the existing karenina
infrastructure code without changing behavior.

Adapter classes:
    - LangChainLLMAdapter: Simple LLM invocation via init_chat_model
    - LangChainAgentAdapter: Agent loops via LangGraph with MCP support
    - LangChainParserAdapter: Structured output parsing via LLM

Utilities:
    - LangChainMessageConverter: Convert between unified Message and LangChain types

Example:
    >>> from karenina.adapters.langchain import LangChainLLMAdapter
    >>> from karenina.schemas.config import ModelConfig
    >>>
    >>> config = ModelConfig(model="claude-sonnet-4-20250514", provider="anthropic")
    >>> adapter = LangChainLLMAdapter(config)
    >>> response = await adapter.ainvoke([Message.user("Hello!")])
"""

from typing import TYPE_CHECKING, Any

# Exports defined for when submodules are implemented
__all__ = [
    # Adapter classes (lc-004, lc-005, lc-006)
    "LangChainLLMAdapter",
    "LangChainAgentAdapter",
    "LangChainParserAdapter",
    # Message conversion (lc-003)
    "LangChainMessageConverter",
    # Trace utilities (lc-009)
    "langchain_messages_to_raw_trace",
    "langchain_messages_to_trace_messages",
    # Usage extraction (lc-007)
    "extract_langchain_usage",
    # Agent utilities
    "extract_partial_agent_state",
]

if TYPE_CHECKING:
    # Type hints for IDE support before implementation
    from karenina.adapters.langchain.agent import (
        LangChainAgentAdapter,
        extract_partial_agent_state,
    )
    from karenina.adapters.langchain.llm import LangChainLLMAdapter
    from karenina.adapters.langchain.messages import LangChainMessageConverter
    from karenina.adapters.langchain.parser import LangChainParserAdapter
    from karenina.adapters.langchain.trace import (
        langchain_messages_to_raw_trace,
        langchain_messages_to_trace_messages,
    )
    from karenina.adapters.langchain.usage import extract_langchain_usage


def __getattr__(name: str) -> Any:
    """Lazy import adapter classes to avoid circular imports.

    This allows importing from karenina.adapters.langchain before all submodules
    are implemented. Each import is resolved on first access.
    """
    if name == "LangChainLLMAdapter":
        from karenina.adapters.langchain.llm import LangChainLLMAdapter

        return LangChainLLMAdapter

    if name == "LangChainAgentAdapter":
        from karenina.adapters.langchain.agent import LangChainAgentAdapter

        return LangChainAgentAdapter

    if name == "LangChainParserAdapter":
        from karenina.adapters.langchain.parser import LangChainParserAdapter

        return LangChainParserAdapter

    if name == "LangChainMessageConverter":
        from karenina.adapters.langchain.messages import LangChainMessageConverter

        return LangChainMessageConverter

    if name == "langchain_messages_to_raw_trace":
        from karenina.adapters.langchain.trace import langchain_messages_to_raw_trace

        return langchain_messages_to_raw_trace

    if name == "langchain_messages_to_trace_messages":
        from karenina.adapters.langchain.trace import langchain_messages_to_trace_messages

        return langchain_messages_to_trace_messages

    if name == "extract_langchain_usage":
        from karenina.adapters.langchain.usage import extract_langchain_usage

        return extract_langchain_usage

    if name == "extract_partial_agent_state":
        from karenina.adapters.langchain.agent import extract_partial_agent_state

        return extract_partial_agent_state

    raise AttributeError(f"module 'karenina.adapters.langchain' has no attribute '{name}'")
