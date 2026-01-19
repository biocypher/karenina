"""Adapter factory for routing to the appropriate LLM backend.

This module provides factory functions that create adapter instances based on
the interface type specified in ModelConfig. The factory handles:

1. **Adapter Selection**: Map `interface` value to correct adapter implementation
2. **Availability Checking**: Detect if required infrastructure is available
3. **Graceful Fallback**: Fall back to LangChain when preferred adapter unavailable
4. **Configuration Conversion**: Convert ModelConfig to adapter-specific options

Supported Interfaces:
    - langchain: Uses LangChainLLMAdapter/LangChainAgentAdapter
    - openrouter: Routes through LangChain adapter
    - openai_endpoint: Routes through LangChain adapter
    - claude_agent_sdk: Uses Claude Agent SDK adapter (when available)
    - manual: Returns None (pre-recorded traces, no adapter needed)

Example:
    >>> from karenina.adapters import get_agent, get_llm, get_parser
    >>> from karenina.schemas.workflow.models import ModelConfig
    >>>
    >>> config = ModelConfig(
    ...     id="claude-sonnet",
    ...     model_name="claude-sonnet-4-20250514",
    ...     model_provider="anthropic"
    ... )
    >>>
    >>> # Get adapters
    >>> agent = get_agent(config)
    >>> llm = get_llm(config)
    >>> parser = get_parser(config)
    >>>
    >>> # Use them
    >>> result = await agent.run(messages=[Message.user("Hello!")])
    >>> response = await llm.ainvoke([Message.user("Hello!")])
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from typing import TYPE_CHECKING

from karenina.ports import (
    AdapterUnavailableError,
    AgentPort,
    LLMPort,
    ParserPort,
)

if TYPE_CHECKING:
    from karenina.schemas.workflow.models import ModelConfig

logger = logging.getLogger(__name__)

# Interface value for Claude Agent SDK
INTERFACE_CLAUDE_AGENT_SDK = "claude_agent_sdk"

# Interfaces that route through LangChain adapter
LANGCHAIN_ROUTED_INTERFACES = frozenset({"langchain", "openrouter", "openai_endpoint"})

# Interface value for manual/pre-recorded traces
INTERFACE_MANUAL = "manual"


@dataclass
class AdapterAvailability:
    """Result of checking adapter availability.

    Attributes:
        available: Whether the adapter can be used.
        reason: Human-readable explanation of availability status.
        fallback_interface: Suggested alternative interface if unavailable.

    Example:
        >>> availability = check_adapter_available("claude_agent_sdk")
        >>> if not availability.available:
        ...     print(f"Claude SDK unavailable: {availability.reason}")
        ...     print(f"Suggested fallback: {availability.fallback_interface}")
    """

    available: bool
    reason: str
    fallback_interface: str | None = None


def check_adapter_available(interface: str) -> AdapterAvailability:
    """Check if an adapter is available for the given interface.

    This function verifies that the required infrastructure for an adapter
    is properly installed and configured.

    Args:
        interface: The interface type to check (e.g., "langchain", "claude_agent_sdk").

    Returns:
        AdapterAvailability with status and details.

    Example:
        >>> result = check_adapter_available("claude_agent_sdk")
        >>> if result.available:
        ...     print("Claude SDK is ready to use")
        ... else:
        ...     print(f"Not available: {result.reason}")
    """
    if interface in LANGCHAIN_ROUTED_INTERFACES:
        # LangChain adapter - check if langchain is importable
        try:
            import langchain_core  # noqa: F401

            return AdapterAvailability(
                available=True,
                reason="LangChain is installed and available",
            )
        except ImportError:
            return AdapterAvailability(
                available=False,
                reason="LangChain packages not installed. Install with: pip install langchain-core langchain-anthropic",
                fallback_interface=None,  # No fallback if LangChain unavailable
            )

    if interface == INTERFACE_CLAUDE_AGENT_SDK:
        # Claude Agent SDK - check if claude CLI is installed
        # The SDK requires the Claude Code CLI to be available in PATH
        claude_path = shutil.which("claude")
        if claude_path is not None:
            return AdapterAvailability(
                available=True,
                reason=f"Claude CLI found at: {claude_path}",
            )
        else:
            return AdapterAvailability(
                available=False,
                reason="Claude Code CLI not found in PATH. Install from: https://claude.ai/code",
                fallback_interface="langchain",
            )

    if interface == INTERFACE_MANUAL:
        # Manual interface always available - uses pre-recorded traces
        return AdapterAvailability(
            available=True,
            reason="Manual interface uses pre-recorded traces",
        )

    # Unknown interface
    return AdapterAvailability(
        available=False,
        reason=f"Unknown interface: {interface}. Supported: langchain, openrouter, openai_endpoint, claude_agent_sdk, manual",
        fallback_interface="langchain",
    )


def get_llm(
    model_config: ModelConfig,
    *,
    auto_fallback: bool = True,
) -> LLMPort | None:
    """Create an LLM adapter for the given model configuration.

    This factory function returns the appropriate LLMPort implementation
    based on the interface specified in the model configuration.

    Args:
        model_config: Configuration specifying model, provider, and interface.
        auto_fallback: If True, automatically fall back to an alternative adapter
            when the preferred one is unavailable. If False, raise an error.

    Returns:
        An LLMPort implementation, or None if interface is "manual".

    Raises:
        AdapterUnavailableError: If the adapter is unavailable and auto_fallback=False.

    Example:
        >>> config = ModelConfig(
        ...     id="claude-sonnet",
        ...     model_name="claude-sonnet-4-20250514",
        ...     model_provider="anthropic"
        ... )
        >>> llm = get_llm(config)
        >>> response = await llm.ainvoke([Message.user("Hello!")])
    """
    interface = model_config.interface

    # Manual interface - no adapter needed
    if interface == INTERFACE_MANUAL:
        return None

    # Check availability
    availability = check_adapter_available(interface)

    if not availability.available:
        if auto_fallback and availability.fallback_interface:
            logger.warning(
                f"Adapter for interface '{interface}' unavailable: {availability.reason}. "
                f"Falling back to '{availability.fallback_interface}'."
            )
            # Create a modified config with fallback interface
            # We need to handle this without modifying the original config
            return _create_llm_adapter(model_config, availability.fallback_interface)
        else:
            raise AdapterUnavailableError(
                message=f"Adapter for interface '{interface}' is not available",
                reason=availability.reason,
                fallback_interface=availability.fallback_interface,
            )

    return _create_llm_adapter(model_config, interface)


def _create_llm_adapter(model_config: ModelConfig, interface: str) -> LLMPort:
    """Create the actual LLM adapter for the given interface.

    Args:
        model_config: Model configuration.
        interface: The interface to use (may differ from config if fallback).

    Returns:
        LLMPort implementation.
    """
    if interface in LANGCHAIN_ROUTED_INTERFACES:
        from karenina.adapters.langchain.llm import LangChainLLMAdapter

        return LangChainLLMAdapter(model_config)

    if interface == INTERFACE_CLAUDE_AGENT_SDK:
        from karenina.adapters.claude_agent_sdk import ClaudeSDKLLMAdapter

        return ClaudeSDKLLMAdapter(model_config)

    # Should not reach here due to availability check
    raise AdapterUnavailableError(
        message=f"No LLM adapter implementation for interface: {interface}",
        reason=f"Unknown interface type: {interface}",
        fallback_interface="langchain",
    )


def get_agent(
    model_config: ModelConfig,
    *,
    auto_fallback: bool = True,
) -> AgentPort | None:
    """Create an Agent adapter for the given model configuration.

    This factory function returns the appropriate AgentPort implementation
    based on the interface specified in the model configuration. Agent adapters
    support tool use and MCP server integration.

    Args:
        model_config: Configuration specifying model, provider, and interface.
        auto_fallback: If True, automatically fall back to an alternative adapter
            when the preferred one is unavailable. If False, raise an error.

    Returns:
        An AgentPort implementation, or None if interface is "manual".

    Raises:
        AdapterUnavailableError: If the adapter is unavailable and auto_fallback=False.

    Example:
        >>> config = ModelConfig(
        ...     id="claude-sonnet",
        ...     model_name="claude-sonnet-4-20250514",
        ...     model_provider="anthropic"
        ... )
        >>> agent = get_agent(config)
        >>> result = await agent.run(
        ...     messages=[Message.user("What files are in /tmp?")],
        ...     mcp_servers={"filesystem": {"type": "http", "url": "http://localhost:8080"}}
        ... )
    """
    interface = model_config.interface

    # Manual interface - no adapter needed
    if interface == INTERFACE_MANUAL:
        return None

    # Check availability
    availability = check_adapter_available(interface)

    if not availability.available:
        if auto_fallback and availability.fallback_interface:
            logger.warning(
                f"Agent adapter for interface '{interface}' unavailable: {availability.reason}. "
                f"Falling back to '{availability.fallback_interface}'."
            )
            return _create_agent_adapter(model_config, availability.fallback_interface)
        else:
            raise AdapterUnavailableError(
                message=f"Agent adapter for interface '{interface}' is not available",
                reason=availability.reason,
                fallback_interface=availability.fallback_interface,
            )

    return _create_agent_adapter(model_config, interface)


def _create_agent_adapter(model_config: ModelConfig, interface: str) -> AgentPort:
    """Create the actual Agent adapter for the given interface.

    Args:
        model_config: Model configuration.
        interface: The interface to use (may differ from config if fallback).

    Returns:
        AgentPort implementation.
    """
    if interface in LANGCHAIN_ROUTED_INTERFACES:
        from karenina.adapters.langchain.agent import LangChainAgentAdapter

        return LangChainAgentAdapter(model_config)

    if interface == INTERFACE_CLAUDE_AGENT_SDK:
        from karenina.adapters.claude_agent_sdk import ClaudeSDKAgentAdapter

        return ClaudeSDKAgentAdapter(model_config)

    # Should not reach here due to availability check
    raise AdapterUnavailableError(
        message=f"No Agent adapter implementation for interface: {interface}",
        reason=f"Unknown interface type: {interface}",
        fallback_interface="langchain",
    )


def get_parser(
    model_config: ModelConfig,
    *,
    auto_fallback: bool = True,
) -> ParserPort | None:
    """Create a Parser adapter for the given model configuration.

    This factory function returns the appropriate ParserPort implementation
    based on the interface specified in the model configuration. Parser adapters
    use LLMs to extract structured data from natural language responses.

    Args:
        model_config: Configuration specifying model, provider, and interface.
        auto_fallback: If True, automatically fall back to an alternative adapter
            when the preferred one is unavailable. If False, raise an error.

    Returns:
        A ParserPort implementation, or None if interface is "manual".

    Raises:
        AdapterUnavailableError: If the adapter is unavailable and auto_fallback=False.

    Example:
        >>> from pydantic import BaseModel, Field
        >>>
        >>> class Answer(BaseModel):
        ...     gene: str = Field(description="Gene name mentioned")
        >>>
        >>> config = ModelConfig(
        ...     id="claude-sonnet",
        ...     model_name="claude-sonnet-4-20250514",
        ...     model_provider="anthropic"
        ... )
        >>> parser = get_parser(config)
        >>> answer = await parser.aparse_to_pydantic(trace_text, Answer)
    """
    interface = model_config.interface

    # Manual interface - no adapter needed
    if interface == INTERFACE_MANUAL:
        return None

    # Check availability
    availability = check_adapter_available(interface)

    if not availability.available:
        if auto_fallback and availability.fallback_interface:
            logger.warning(
                f"Parser adapter for interface '{interface}' unavailable: {availability.reason}. "
                f"Falling back to '{availability.fallback_interface}'."
            )
            return _create_parser_adapter(model_config, availability.fallback_interface)
        else:
            raise AdapterUnavailableError(
                message=f"Parser adapter for interface '{interface}' is not available",
                reason=availability.reason,
                fallback_interface=availability.fallback_interface,
            )

    return _create_parser_adapter(model_config, interface)


def _create_parser_adapter(model_config: ModelConfig, interface: str) -> ParserPort:
    """Create the actual Parser adapter for the given interface.

    Args:
        model_config: Model configuration.
        interface: The interface to use (may differ from config if fallback).

    Returns:
        ParserPort implementation.
    """
    if interface in LANGCHAIN_ROUTED_INTERFACES:
        from karenina.adapters.langchain.parser import LangChainParserAdapter

        return LangChainParserAdapter(model_config)

    if interface == INTERFACE_CLAUDE_AGENT_SDK:
        from karenina.adapters.claude_agent_sdk import ClaudeSDKParserAdapter

        return ClaudeSDKParserAdapter(model_config)

    # Should not reach here due to availability check
    raise AdapterUnavailableError(
        message=f"No Parser adapter implementation for interface: {interface}",
        reason=f"Unknown interface type: {interface}",
        fallback_interface="langchain",
    )
