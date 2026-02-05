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
    - manual: Returns ManualAdapter (raises error if invoked)

Example:
    >>> from karenina.adapters import get_agent, get_llm, get_parser
    >>> from karenina.schemas.config import ModelConfig
    >>>
    >>> config = ModelConfig(
    ...     id="claude-sonnet",
    ...     model_name="claude-sonnet-4-20250514",
    ...     model_provider="anthropic"
    ... )
    >>>
    >>> # Get adapters (always returns a port, never None)
    >>> agent = get_agent(config)
    >>> llm = get_llm(config)
    >>> parser = get_parser(config)
    >>>
    >>> # Use them (check interface before using for manual)
    >>> if config.interface != "manual":
    ...     result = await agent.arun(messages=[Message.user("Hello!")])
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, cast

from karenina.adapters.registry import AdapterAvailability, AdapterRegistry, register_adapter
from karenina.ports import (
    AdapterUnavailableError,
    AgentPort,
    LLMPort,
    ParserPort,
)

if TYPE_CHECKING:
    from karenina.schemas.config import ModelConfig

# Import at runtime for validation function (not circular since ModelConfig is only TYPE_CHECKING)
from karenina.schemas.config import INTERFACES_NO_PROVIDER_REQUIRED

# Type alias for interface values
InterfaceType: TypeAlias = Literal[
    "langchain", "openrouter", "manual", "openai_endpoint", "claude_agent_sdk", "claude_tool"
]

logger = logging.getLogger(__name__)

# Interface value for Claude Agent SDK
INTERFACE_CLAUDE_AGENT_SDK = "claude_agent_sdk"

# Interfaces that route through LangChain adapter
LANGCHAIN_ROUTED_INTERFACES = frozenset({"langchain", "openrouter", "openai_endpoint"})

# Interface value for manual/pre-recorded traces
INTERFACE_MANUAL = "manual"


def validate_model_config(model_config: ModelConfig | None) -> None:
    """Validate that a model configuration has all required fields.

    This centralizes validation that was previously duplicated across evaluators.
    Called by factory functions before creating adapters.

    Args:
        model_config: The model configuration to validate.

    Raises:
        ValueError: If model_config is None, model_name is empty, or
            model_provider is missing for interfaces that require it.

    Example:
        >>> config = ModelConfig(model_name="gpt-4", model_provider="openai")
        >>> validate_model_config(config)  # OK
        >>>
        >>> validate_model_config(None)  # Raises ValueError
    """
    if not model_config:
        raise AdapterUnavailableError("Model configuration is required", reason="missing_config")

    if not model_config.model_name:
        raise AdapterUnavailableError("Model name is required in model configuration", reason="missing_model_name")

    # Model provider is optional for OpenRouter, manual, and openai_endpoint interfaces
    if model_config.interface not in INTERFACES_NO_PROVIDER_REQUIRED and not model_config.model_provider:
        raise AdapterUnavailableError(
            f"Model provider is required for interface '{model_config.interface}'. "
            f"Only {list(INTERFACES_NO_PROVIDER_REQUIRED)} interfaces allow empty providers.",
            reason="missing_model_provider",
        )


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
    return AdapterRegistry.check_availability(interface)


def get_llm(
    model_config: ModelConfig,
    *,
    auto_fallback: bool = True,
) -> LLMPort:
    """Create an LLM adapter for the given model configuration.

    This factory function returns the appropriate LLMPort implementation
    based on the interface specified in the model configuration.

    IMPORTANT: This function always returns an LLMPort, never None.
    For manual interface, returns ManualLLMAdapter which raises
    ManualInterfaceError if invoked. Call sites should check
    `model_config.interface != "manual"` before using the adapter.

    Args:
        model_config: Configuration specifying model, provider, and interface.
        auto_fallback: If True, automatically fall back to an alternative adapter
            when the preferred one is unavailable. If False, raise an error.

    Returns:
        An LLMPort implementation.

    Raises:
        AdapterUnavailableError: If the adapter is unavailable and auto_fallback=False.

    Example:
        >>> config = ModelConfig(
        ...     id="claude-sonnet",
        ...     model_name="claude-sonnet-4-20250514",
        ...     model_provider="anthropic"
        ... )
        >>> llm = get_llm(config)
        >>> if config.interface != "manual":
        ...     response = await llm.ainvoke([Message.user("Hello!")])
    """
    # Validate config has required fields
    validate_model_config(model_config)

    interface = model_config.interface

    # Check availability
    availability = AdapterRegistry.check_availability(interface)

    if not availability.available:
        if auto_fallback and availability.fallback_interface:
            logger.warning(
                f"Adapter for interface '{interface}' unavailable: {availability.reason}. "
                f"Falling back to '{availability.fallback_interface}'."
            )
            interface = cast(InterfaceType, availability.fallback_interface)
            # Transform the config so the adapter only sees interfaces it handles
            model_config = model_config.model_copy(update={"interface": interface})
        else:
            raise AdapterUnavailableError(
                message=f"Adapter for interface '{interface}' is not available",
                reason=availability.reason,
                fallback_interface=availability.fallback_interface,
            )

    # Get adapter spec from registry
    spec = AdapterRegistry.get_spec(interface)
    if spec is None or spec.llm_factory is None:
        raise AdapterUnavailableError(
            message=f"No LLM adapter implementation for interface: {interface}",
            reason=f"Interface '{interface}' does not have an LLM factory registered",
            fallback_interface="langchain",
        )

    adapter = spec.llm_factory(model_config)
    if not isinstance(adapter, LLMPort):
        raise AdapterUnavailableError(
            message=f"LLM adapter for interface '{interface}' does not implement LLMPort protocol",
            reason=f"Adapter {type(adapter).__name__} is missing required methods",
        )
    register_adapter(adapter)
    return adapter


def get_agent(
    model_config: ModelConfig,
    *,
    auto_fallback: bool = True,
) -> AgentPort:
    """Create an Agent adapter for the given model configuration.

    This factory function returns the appropriate AgentPort implementation
    based on the interface specified in the model configuration. Agent adapters
    support tool use and MCP server integration.

    IMPORTANT: This function always returns an AgentPort, never None.
    For manual interface, returns ManualAgentAdapter which raises
    ManualInterfaceError if invoked. Call sites should check
    `model_config.interface != "manual"` before using the adapter.

    Args:
        model_config: Configuration specifying model, provider, and interface.
        auto_fallback: If True, automatically fall back to an alternative adapter
            when the preferred one is unavailable. If False, raise an error.

    Returns:
        An AgentPort implementation.

    Raises:
        AdapterUnavailableError: If the adapter is unavailable and auto_fallback=False.

    Example:
        >>> config = ModelConfig(
        ...     id="claude-sonnet",
        ...     model_name="claude-sonnet-4-20250514",
        ...     model_provider="anthropic"
        ... )
        >>> agent = get_agent(config)
        >>> if config.interface != "manual":
        ...     result = await agent.arun(
        ...         messages=[Message.user("What files are in /tmp?")],
        ...         mcp_servers={"filesystem": {"type": "http", "url": "http://localhost:8080"}}
        ...     )
    """
    # Validate config has required fields
    validate_model_config(model_config)

    interface = model_config.interface

    # Check availability
    availability = AdapterRegistry.check_availability(interface)

    if not availability.available:
        if auto_fallback and availability.fallback_interface:
            logger.warning(
                f"Agent adapter for interface '{interface}' unavailable: {availability.reason}. "
                f"Falling back to '{availability.fallback_interface}'."
            )
            interface = cast(InterfaceType, availability.fallback_interface)
            # Transform the config so the adapter only sees interfaces it handles
            model_config = model_config.model_copy(update={"interface": interface})
        else:
            raise AdapterUnavailableError(
                message=f"Agent adapter for interface '{interface}' is not available",
                reason=availability.reason,
                fallback_interface=availability.fallback_interface,
            )

    # Get adapter spec from registry
    spec = AdapterRegistry.get_spec(interface)
    if spec is None or spec.agent_factory is None:
        raise AdapterUnavailableError(
            message=f"No Agent adapter implementation for interface: {interface}",
            reason=f"Interface '{interface}' does not have an agent factory registered",
            fallback_interface="langchain",
        )

    adapter = spec.agent_factory(model_config)
    if not isinstance(adapter, AgentPort):
        raise AdapterUnavailableError(
            message=f"Agent adapter for interface '{interface}' does not implement AgentPort protocol",
            reason=f"Adapter {type(adapter).__name__} is missing required methods",
        )
    register_adapter(adapter)
    return adapter


def get_parser(
    model_config: ModelConfig,
    *,
    auto_fallback: bool = True,
) -> ParserPort:
    """Create a Parser adapter for the given model configuration.

    This factory function returns the appropriate ParserPort implementation
    based on the interface specified in the model configuration. Parser adapters
    use LLMs to extract structured data from natural language responses.

    IMPORTANT: This function always returns a ParserPort, never None.
    For manual interface, returns ManualParserAdapter which raises
    ManualInterfaceError if invoked. Call sites should check
    `model_config.interface != "manual"` before using the adapter.

    Args:
        model_config: Configuration specifying model, provider, and interface.
        auto_fallback: If True, automatically fall back to an alternative adapter
            when the preferred one is unavailable. If False, raise an error.

    Returns:
        A ParserPort implementation.

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
        >>> if config.interface != "manual":
        ...     answer = await parser.aparse_to_pydantic(trace_text, Answer)
    """
    # Validate config has required fields
    validate_model_config(model_config)

    interface = model_config.interface

    # Check availability
    availability = AdapterRegistry.check_availability(interface)

    if not availability.available:
        if auto_fallback and availability.fallback_interface:
            logger.warning(
                f"Parser adapter for interface '{interface}' unavailable: {availability.reason}. "
                f"Falling back to '{availability.fallback_interface}'."
            )
            interface = cast(InterfaceType, availability.fallback_interface)
            # Transform the config so the adapter only sees interfaces it handles
            model_config = model_config.model_copy(update={"interface": interface})
        else:
            raise AdapterUnavailableError(
                message=f"Parser adapter for interface '{interface}' is not available",
                reason=availability.reason,
                fallback_interface=availability.fallback_interface,
            )

    # Get adapter spec from registry
    spec = AdapterRegistry.get_spec(interface)
    if spec is None or spec.parser_factory is None:
        raise AdapterUnavailableError(
            message=f"No Parser adapter implementation for interface: {interface}",
            reason=f"Interface '{interface}' does not have a parser factory registered",
            fallback_interface="langchain",
        )

    adapter = spec.parser_factory(model_config)
    if not isinstance(adapter, ParserPort):
        raise AdapterUnavailableError(
            message=f"Parser adapter for interface '{interface}' does not implement ParserPort protocol",
            reason=f"Adapter {type(adapter).__name__} is missing required methods",
        )
    register_adapter(adapter)
    return adapter


def build_llm_kwargs(
    model_config: ModelConfig,
    *,
    question_hash: str | None = None,
) -> dict[str, Any]:
    """Build kwargs dict for init_chat_model_unified from a ModelConfig.

    This centralizes the interface-specific parameter handling that was previously
    duplicated across call sites (generate_answer.py, template_evaluator.py, etc.).

    Handles:
    - Base parameters (model, provider, temperature, interface)
    - OpenAI endpoint configuration (base_url, api_key)
    - Manual interface (question_hash)
    - MCP server configuration (urls, tool filter, description overrides)
    - Agent middleware configuration
    - Max context tokens
    - Extra kwargs from model config

    Args:
        model_config: Model configuration with all settings.
        question_hash: MD5 hash of the question (required for manual interface).

    Returns:
        Dict of kwargs ready to pass to init_chat_model_unified().

    Raises:
        ValueError: If required parameters are missing for the interface.

    Example:
        >>> config = ModelConfig(
        ...     model_name="gpt-4",
        ...     model_provider="openai",
        ...     interface="langchain"
        ... )
        >>> kwargs = build_llm_kwargs(config)
        >>> llm = init_chat_model_unified(**kwargs)
    """
    kwargs: dict[str, Any] = {
        "model": model_config.model_name,
        "provider": model_config.model_provider,
        "temperature": model_config.temperature,
        "interface": model_config.interface,
    }

    # Add MCP configuration if present
    if model_config.mcp_urls_dict:
        kwargs["mcp_urls_dict"] = model_config.mcp_urls_dict
    if model_config.mcp_tool_filter:
        kwargs["mcp_tool_filter"] = model_config.mcp_tool_filter
    if model_config.mcp_tool_description_overrides:
        kwargs["mcp_tool_description_overrides"] = model_config.mcp_tool_description_overrides

    # Agent middleware config (for MCP-enabled agents)
    if model_config.agent_middleware is not None:
        kwargs["agent_middleware_config"] = model_config.agent_middleware

    # Max context tokens (for summarization middleware)
    if model_config.max_context_tokens is not None:
        kwargs["max_context_tokens"] = model_config.max_context_tokens

    # Interface-specific parameters
    if model_config.interface == "openai_endpoint":
        if not model_config.endpoint_base_url:
            raise AdapterUnavailableError(
                "endpoint_base_url is required for openai_endpoint interface",
                reason="missing_endpoint_base_url",
            )
        if not model_config.endpoint_api_key:
            raise AdapterUnavailableError(
                "endpoint_api_key is required for openai_endpoint interface",
                reason="missing_endpoint_api_key",
            )

        kwargs["endpoint_base_url"] = model_config.endpoint_base_url
        kwargs["endpoint_api_key"] = model_config.endpoint_api_key.get_secret_value()

    elif model_config.interface == "manual":
        if question_hash is not None:
            kwargs["question_hash"] = question_hash

    # Add extra kwargs if provided
    if model_config.extra_kwargs:
        kwargs.update(model_config.extra_kwargs)

    return kwargs
