"""Native tool-calling agents for karenina.

This package provides native OpenAI and Anthropic tool-calling agent implementations
that bypass LangChain/LangGraph while maintaining compatibility with MCP tools.

Usage:
    from karenina.infrastructure.llm.native_agents import create_native_agent

    # Create agent with LangChain MCP tools
    agent = create_native_agent(
        provider="openai",
        model="gpt-4.1-mini",
        tools=mcp_tools,  # LangChain tools from langchain-mcp-adapters
        temperature=0.0,
    )

    # Invoke agent
    response = await agent.ainvoke([{"role": "user", "content": "What is..."}])
"""

from __future__ import annotations

from typing import Any

from .base import NativeAgentBase, NativeAgentResponse
from .tool_converter import NativeToolConverter


def create_native_agent(
    provider: str,
    model: str,
    tools: list[Any],
    system_prompt: str | None = None,
    temperature: float = 0.0,
    max_iterations: int = 10,
    api_key: str | None = None,
    **kwargs: Any,
) -> NativeAgentBase:
    """Factory function to create a native tool-calling agent.

    Creates an OpenAI or Anthropic native agent that uses the respective SDK
    directly for tool calling instead of going through LangChain/LangGraph.

    Args:
        provider: Provider name - "openai", "openrouter", or "anthropic"
        model: Model name/identifier (e.g., "gpt-4.1-mini", "claude-sonnet-4-20250514")
        tools: List of LangChain MCP tools (from langchain-mcp-adapters)
        system_prompt: Optional system prompt
        temperature: Model temperature (default 0.0)
        max_iterations: Maximum agent loop iterations (default 10)
        api_key: Optional explicit API key (falls back to env vars)
        **kwargs: Additional model parameters (e.g., max_tokens)

    Returns:
        NativeAgentBase instance (OpenAINativeAgent or AnthropicNativeAgent)

    Raises:
        ValueError: If provider is not supported
        ImportError: If the required SDK is not installed

    Example:
        >>> from langchain_mcp_adapters.client import MultiServerMCPClient
        >>> # Fetch MCP tools
        >>> client = MultiServerMCPClient(server_config)
        >>> tools = await client.get_tools()
        >>> # Create native agent
        >>> agent = create_native_agent(
        ...     provider="openai",
        ...     model="gpt-4.1-mini",
        ...     tools=tools,
        ...     temperature=0.0,
        ...     max_iterations=10,
        ... )
        >>> # Run agent
        >>> response = await agent.ainvoke([{"role": "user", "content": "..."}])
    """
    # Convert tools to native format based on provider
    if provider in ("openai", "openrouter"):
        from .openai_agent import OpenAINativeAgent

        native_tools, tool_map = NativeToolConverter.to_openai(tools)
        return OpenAINativeAgent(
            model=model,
            tools=native_tools,
            tool_map=tool_map,
            system_prompt=system_prompt,
            temperature=temperature,
            max_iterations=max_iterations,
            api_key=api_key,
            **kwargs,
        )
    elif provider == "anthropic":
        from .anthropic_agent import AnthropicNativeAgent

        native_tools, tool_map = NativeToolConverter.to_anthropic(tools)
        return AnthropicNativeAgent(
            model=model,
            tools=native_tools,
            tool_map=tool_map,
            system_prompt=system_prompt,
            temperature=temperature,
            max_iterations=max_iterations,
            api_key=api_key,
            **kwargs,
        )
    else:
        raise ValueError(
            f"Native tool calling not supported for provider: {provider}. "
            f"Supported providers: openai, openrouter, anthropic"
        )


__all__ = [
    "create_native_agent",
    "NativeAgentBase",
    "NativeAgentResponse",
    "NativeToolConverter",
]
